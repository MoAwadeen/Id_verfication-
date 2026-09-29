from supabase import create_client, Client
from id_image import IDResult
import asyncio
from concurrent.futures import ThreadPoolExecutor


class SupabaseClient:
    def __init__(self, url: str, key: str, executor: ThreadPoolExecutor):
        self.client: Client = create_client(url, key)
        self._executor = executor

    async def save_id_result(self, result: IDResult) -> tuple[bool, str]:
        def _insert():
            data = {k: v for k, v in {
                "id_number": result.id_number,
                "first_name": result.first_name,
                "full_name": result.full_name,
                "year": result.year,
                "month": result.month,
                "day": result.day,
                "birth_date": result.birth_date,
                "governorate": result.governorate,
                "gender": result.gender,
            }.items() if v is not None}
            self.client.table("id_results").insert(data).execute()
            return True, "Saved successfully"

        try:
            return await asyncio.get_event_loop().run_in_executor(self._executor, _insert)
        except Exception as e:
            error_msg = str(e)
            if "duplicate key value violates unique constraint" in error_msg:
                return False, f"ID already exists: {result.id_number}"
            return False, f"Error: {error_msg}"